#!/bin/bash
# Real perception services for the eval-only realistic path (SOARM_PERCEPTION=real):
#   tools/serve_perception.sh sam3|depth|molmo|all <gpu>      (stop: kill $(cat $LOG_DIR/<name>.pid))
#   sam3  :8114  harness services/sam3_server.py                       ($PERCEP_PYTHON with capx + sam3, $SAM3_CHECKPOINT)
#                SAM3_CACHE_IMAGES (default: the SOARM_PROFILE's services.sam3_cache_images, else 0 = off) -> its
#                --cache-images image-encoder cache (same responses)
#   depth :8130  MoGe-2 ViT-L, /estimate_depth protocol of mono_depth.py ($DEPTH_SERVER, default harness
#                services/depth_server.py; MOGE_CHECKPOINT = the MoGe-2 model.pt)
#                DEPTH_PORT [8130], DEPTH_RESOLUTION_LEVEL [default: the SOARM_PROFILE's services.depth_resolution_level,
#                else the server's default 9, the maximum token budget;
#                0 halves the ViT-L forward, 0.9 % median depth change]; a second server on
#                another port is reached per episode with CAPTAIN_DEPTH_URL
#   molmo :8122  Molmo2-8B (rev e28fa28, bf16 weights) in vLLM with online FP8 weights: fits one 24 GB 4090
#                (bf16 needs ~29.4 GiB at the release flags). MOLMO_ARGS overrides the quantization flags.
source "$(cd "$(dirname "$0")/.." && pwd)/env.sh"
WHAT=${1:?sam3|depth|molmo|all}; G=${2:?gpu}
PERCEP_PYTHON=${PERCEP_PYTHON:-$SOARM_ROOT/venv-percep/bin/python}
SAM3_CHECKPOINT=${SAM3_CHECKPOINT:-$SOARM_ROOT/weights/sam3/sam3.pt}
DEPTH_SERVER=${DEPTH_SERVER:-$HARNESS_DIR/services/depth_server.py}
MOLMO_MODEL=${MOLMO_MODEL:-$SOARM_ROOT/speed/weights/Molmo2-8B}
start() {  # name cmd...
  local n=$1; shift
  [ -f "$LOG_DIR/$n.pid" ] && kill "$(cat "$LOG_DIR/$n.pid")" 2>/dev/null
  CUDA_VISIBLE_DEVICES=$G setsid nohup "$@" > "$LOG_DIR/$n.log" 2>&1 < /dev/null &
  echo $! > "$LOG_DIR/$n.pid"; echo "$n pid=$! gpu=$G log=$LOG_DIR/$n.log"
}
if [ "$WHAT" = sam3 ] || [ "$WHAT" = all ]; then
  CACHE=${SAM3_CACHE_IMAGES:-0}
  if [ -z "${SAM3_CACHE_IMAGES:-}" ] && [ -n "${SOARM_PROFILE:-}" ]; then
    CACHE=$(python -m soarm.speed_profile sam3-cache-images "$SOARM_PROFILE") || exit 2
  fi
  CACHE_ARGS=()
  [ "$CACHE" -gt 0 ] && CACHE_ARGS=(--cache-images "$CACHE")
  start sam3 "$PERCEP_PYTHON" "$HARNESS_DIR/services/sam3_server.py" --port 8114 --checkpoint "$SAM3_CHECKPOINT" \
    "${CACHE_ARGS[@]}"
fi
if [ "$WHAT" = depth ] || [ "$WHAT" = all ]; then
  LEVEL=${DEPTH_RESOLUTION_LEVEL:-}
  if [ -z "$LEVEL" ] && [ -n "${SOARM_PROFILE:-}" ]; then
    LEVEL=$(python -m soarm.speed_profile depth-resolution-level "$SOARM_PROFILE") || exit 2
  fi
  LEVEL_ARGS=()
  [ -n "$LEVEL" ] && LEVEL_ARGS=(--resolution-level "$LEVEL")
  NAME=depth; [ "${DEPTH_PORT:-8130}" != 8130 ] && NAME=depth_${DEPTH_PORT}
  (cd "$(dirname "$(dirname "$DEPTH_SERVER")")" && HF_HUB_OFFLINE=1 start "$NAME" "$PERCEP_PYTHON" "$DEPTH_SERVER" \
    --port "${DEPTH_PORT:-8130}" ${MOGE_CHECKPOINT:+--checkpoint "$MOGE_CHECKPOINT"} "${LEVEL_ARGS[@]}")
fi
if [ "$WHAT" = molmo ] || [ "$WHAT" = all ]; then
  VLLM_USE_FLASHINFER_SAMPLER=0 start molmo "$VLLM_VENV/bin/vllm" serve "$MOLMO_MODEL" --served-model-name allenai/Molmo2-8B Molmo2-8B molmo \
    --host 127.0.0.1 --port 8122 --trust-remote-code --dtype bfloat16 --max-model-len 16384 \
    --max-num-batched-tokens 16384 --limit-mm-per-prompt '{"image":2,"video":0}' --max-num-seqs 2 \
    --gpu-memory-utilization ${MOLMO_MEM:-0.92} ${MOLMO_ARGS:---quantization fp8}
fi

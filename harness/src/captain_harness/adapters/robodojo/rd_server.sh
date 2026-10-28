#!/bin/bash
# rd_server.sh <rfd> <wfd> : start the RoboDojo scene server under RoboDojo's Isaac python (env from $RD_ENV, cwd $RD_ROOT)
# RD_GPU_PHYS=<n>: confine Isaac to that one GPU. CUDA_VISIBLE_DEVICES masks CUDA, and H100FIX_ONLY_UUID makes the optional h100fix
# layer hide every other Vulkan device, so CUDA and Vulkan both enumerate the same single GPU (index 0 in the process).
# Both are needed: the mask alone leaves Vulkan on all GPUs and breaks Omniverse's CUDA<->Vulkan device mapping.
source "${RD_ENV:?}"
cd "${RD_ROOT:?}"
if [ -n "${RD_GPU_PHYS:-}" ] && [ "${RD_CONFINE_MODE:-both}" != none ]; then
  M=${RD_CONFINE_MODE:-both}   # both (default) | vk | cuda (diagnostics)
  if [ "$M" != cuda ]; then
    export H100FIX_ONLY_UUID=$(nvidia-smi -i "$RD_GPU_PHYS" --query-gpu=uuid --format=csv,noheader)
    [ -n "$H100FIX_ONLY_UUID" ] || { echo "rd_server.sh: no uuid for GPU $RD_GPU_PHYS" >&2; exit 3; }
  fi
  [ "$M" != vk ] && export CUDA_VISIBLE_DEVICES=$RD_GPU_PHYS
fi
# own temp dir: /tmp/isaaclab may belong to another user on shared nodes (IsaacLab logs there)
export TMPDIR=${RD_TMPDIR:-$HOME/robodojo/tmp}; mkdir -p "$TMPDIR"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
exec python -u "$(dirname "$(readlink -f "$0")")/rd_server.py" "$@"

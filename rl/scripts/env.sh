# Common environment of the CaPtain RL scripts (sourced). Override any of these before calling a script.
#   CAPTAIN_ROOT       the CaPtain checkout (contains harness/ and rl/)            [two levels above this file]
#   CAPTAIN_WORKSPACE  runs, envs, models, critic, logs (shared by all nodes)      [$HOME/captain]
#   CAPTAIN_PY         python with the rollout-side deps (fastapi, uvicorn, requests, numpy, torch-cpu ok) [$CAPTAIN_WORKSPACE/envs/libero/bin/python]
#   CAPTAIN_VLLM_PY    python of the vLLM environment used for the policy servers   [$CAPTAIN_WORKSPACE/envs/vllm/bin/python]
#   CAPTAIN_SSH        ssh command used by run_stream.sh to reach other nodes       [ssh -o BatchMode=yes]
: "${CAPTAIN_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
: "${CAPTAIN_WORKSPACE:=$HOME/captain}"
: "${CAPTAIN_PY:=$CAPTAIN_WORKSPACE/envs/libero/bin/python}"
: "${CAPTAIN_VLLM_PY:=$CAPTAIN_WORKSPACE/envs/vllm/bin/python}"
: "${CAPTAIN_SSH:=ssh -o BatchMode=yes}"
: "${CAPTAIN_LOGS:=$CAPTAIN_WORKSPACE/logs}"
export CAPTAIN_ROOT CAPTAIN_WORKSPACE CAPTAIN_PY CAPTAIN_VLLM_PY CAPTAIN_SSH CAPTAIN_LOGS
# local services must never go through an HTTP proxy
export no_proxy="127.0.0.1,localhost${no_proxy:+,$no_proxy}" NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
# EGL rendering of the LIBERO env servers (NVIDIA glvnd ICD)
export __EGL_VENDOR_LIBRARY_FILENAMES=${__EGL_VENDOR_LIBRARY_FILENAMES:-/usr/share/glvnd/egl_vendor.d/10_nvidia.json}
mkdir -p "$CAPTAIN_LOGS"
# read one value of a run config (with ${VAR} expansion, as fresh_run does): cfgget <config.json> <python expression over c>
cfgget() { PYTHONPATH="$CAPTAIN_ROOT/rl" "$CAPTAIN_PY" -c "import sys, fresh_run; c = fresh_run.load_config(sys.argv[1]); print(eval(sys.argv[2]))" "$1" "$2"; }
# trainer environment of the reference run (H100 80 GB; see rl/README.md "Trainer environment")
default_train_env() {
  local nccl_so; nccl_so=$(ls "$(dirname "$CAPTAIN_VLLM_PY")"/../lib/python3*/site-packages/nvidia/nccl/lib/libnccl.so.2 2>/dev/null | head -1)
  echo "RL_KEEP_PREV_FULL_STATE=1 RL_FSDP_CPU_OFFLOAD=0 RL_ACT_OFFLOAD_MIN_TOKENS=1000000 RL_ASYNC_OFFLOAD=1 RL_EMPTY_CACHE=auto RL_REFIT=nccl RL_REFIT_NCCL_SO=$nccl_so RL_INFRA_WARN_ONLY=1 RL_PROFILE=1 RL_ASYNC_CKPT_PROCESS=0 RL_PAYLOAD_NCCL=1 RL_KL_ON_KEPT=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600"
}
